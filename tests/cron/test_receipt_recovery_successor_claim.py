"""Receipt recovery of a dead firer must not clear a newer live firer's claim (#152296).

Uses an isolated HERMES_HOME; touches no production state. A dead firer's receipt
recovery (execution A) must defer while a newer live run (B) holds the job's fire claim,
retain A's journal without delivering or booking A's result, and settle A only after B
finishes through its own owner-fenced terminal write.
"""
from contextlib import asynccontextmanager
import json
import os
import subprocess
import sys

from cron import executions, jobs, scheduler_authority


def test_dead_firer_receipt_recovery_keeps_live_firers_claim(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with jobs.use_cron_store(tmp_path / 'cron'):
        job = jobs.create_job(prompt='report', schedule='0 8 * * 1-5', deliver='local')

        # A's direct CLI firer exits while its admitted execution remains running.
        code = ('from cron.executions import create_execution, mark_execution_running; '
                f'row=create_execution({job["id"]!r},source="direct"); '
                'mark_execution_running(row["id"]); print(row["id"],flush=True)')
        dead = subprocess.run([sys.executable, '-c', code], env=dict(os.environ),
                              capture_output=True, text=True, check=True)
        fire_a = dead.stdout.strip().splitlines()[-1]
        assert executions.get_execution(fire_a)['status'] == 'running'
        params = {'job_id': job['id'], 'request_id': fire_a, 'extra_prompt': None}
        journal = scheduler_authority.journal_path(job['id'], fire_a)
        journal.parent.mkdir(parents=True)
        journal.write_text(json.dumps({'params': params, 'receipt': {'session_id': 's', 'admission_id': 'a'}}),
                           encoding='utf-8')

        # B is a newer live direct run with a fresh claim.
        claimed = jobs.claim_job_for_fire(job['id'], force=True, return_job=True)
        assert claimed, 'fixture: live run B could not claim'
        owner_b = claimed['fire_claim']['by']
        assert jobs.heartbeat_fire_claim(job['id'], expected_owner=owner_b)

        # The canonical owner reports A terminal; a gateway tick reconciles it.
        class Peer:
            async def rpc(self, method, **kwargs):
                assert method == 'cron.recover' and kwargs == params
                return {'status': 'terminal', 'job': jobs.get_job(job['id']),
                        'result': [True, 'report', 'answer', None]}

        @asynccontextmanager
        async def connect():
            yield Peer()

        monkeypatch.setattr('hermes_cli.gateway_client.connect_gateway', connect)
        scheduler_authority.reconcile_pending()

        # Recovering A is not B's terminal completion.
        assert jobs.heartbeat_fire_claim(job['id'], expected_owner=owner_b), (
            'receipt recovery of a dead firer cleared the live firer\'s fire_claim '
            f'(fire_claim now {jobs.get_job(job["id"]).get("fire_claim")!r})')
        # While deferred, A's receipt must produce no output, delivery, or job bookkeeping.
        assert journal.exists()
        assert not list(jobs._job_output_dir(job['id']).glob('*.md'))
        assert jobs.get_job(job['id'])['last_run_at'] is None

        # B finishes through its own owner-fenced terminal write.
        assert jobs.mark_job_run(job['id'], True, expected_fire_owner=owner_b)
        # A's deferred receipt must still settle after B finishes.
        scheduler_authority.reconcile_pending()
        assert executions.get_execution(fire_a)['status'] == 'completed'
        assert not journal.exists()
