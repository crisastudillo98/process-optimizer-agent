"""add_chat_messages_and_bpmn_paths

Revision ID: a3f8d2e1c9b7
Revises: 6d912cc73477
Create Date: 2026-05-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3f8d2e1c9b7'
down_revision: Union[str, Sequence[str], None] = '6d912cc73477'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add BPMN path columns to analyses table
    with op.batch_alter_table('analyses', schema=None) as batch_op:
        batch_op.add_column(sa.Column('bpmn_tobe_path', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('bpmn_asis_path', sa.String(), nullable=True))

    # Create chat_messages table
    op.create_table(
        'chat_messages',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('(CURRENT_TIMESTAMP)'),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(['session_id'], ['analyses.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('chat_messages', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_chat_messages_session_id'), ['session_id'], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table('chat_messages', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_chat_messages_session_id'))
    op.drop_table('chat_messages')

    with op.batch_alter_table('analyses', schema=None) as batch_op:
        batch_op.drop_column('bpmn_asis_path')
        batch_op.drop_column('bpmn_tobe_path')
