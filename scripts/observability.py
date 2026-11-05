"""生成独立本地配置并管理官方Compose；文件名避免遮蔽langfuse官方包。"""

import argparse
from pathlib import Path
import secrets
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def initialize(root=ROOT):
    root = Path(root)
    private = root / '.env.langfuse'
    if not private.exists():
        fields = {
            name: secrets.token_hex(24)
            for name in (
                'LF_POSTGRES_PASSWORD',
                'LF_CLICKHOUSE_PASSWORD',
                'LF_REDIS_PASSWORD',
                'LF_MINIO_PASSWORD',
                'LF_ADMIN_PASSWORD',
                'LF_NEXTAUTH_SECRET',
                'LF_SALT',
            )
        }
        fields.update(
            LF_ENCRYPTION_KEY=secrets.token_hex(32),
            LF_PUBLIC_KEY='pk-lf-' + secrets.token_hex(16),
            LF_SECRET_KEY='sk-lf-' + secrets.token_hex(24),
        )
        private.write_text(
            '\n'.join(f'{k}={v}' for k, v in fields.items()) + '\n', encoding='utf-8'
        )
    else:
        fields = dict(
            line.split('=', 1)
            for line in private.read_text(encoding='utf-8').splitlines()
            if line and not line.startswith('#')
        )
    application = root / '.env.observability'
    if not application.exists():
        application.write_text(
            '\n'.join(
                [
                    'LANGFUSE_ENABLED=true',
                    'LANGFUSE_BASE_URL=http://127.0.0.1:3000',
                    'LANGFUSE_PUBLIC_KEY=' + fields['LF_PUBLIC_KEY'],
                    'LANGFUSE_SECRET_KEY=' + fields['LF_SECRET_KEY'],
                    'LEARNING_FLYWHEEL_ENABLED=true',
                    'LEARNING_EVALUATION_ENABLED=false',
                    'LEARNING_EVALUATION_HOURS=24',
                ]
            )
            + '\n',
            encoding='utf-8',
        )
    print(
        '本地配置已就绪；既有文件保留。登录账号admin@mewhelp.local，密码仅保存在.env.langfuse。'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['init', 'up', 'stop', 'status'])
    command = parser.parse_args().command
    if command == 'init':
        initialize()
    else:
        options = {
            'up': ['up', '-d', '--wait', '--wait-timeout', '300'],
            'stop': ['stop'],
            'status': ['ps'],
        }
        raise SystemExit(
            subprocess.run(
                [
                    'docker',
                    'compose',
                    '--env-file',
                    '.env.langfuse',
                    '-f',
                    'docker-compose.langfuse.yml',
                    *options[command],
                ],
                cwd=ROOT,
            ).returncode
        )
