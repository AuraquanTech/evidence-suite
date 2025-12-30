"""Initial migration - all models

Revision ID: 001_initial
Revises:
Create Date: 2024-12-30

Evidence Suite database schema:
- Cases
- Evidence
- ChainOfCustody
- AnalysisResults
- AnalysisJobs
- Users
- AuditLog
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create users table first (referenced by audit_log)
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('name', sa.String(200), nullable=True),
        sa.Column('role', sa.String(50), nullable=True, default='analyst'),
        sa.Column('password_hash', sa.String(255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True, default=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )

    # Create cases table
    op.create_table(
        'cases',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('case_number', sa.String(100), nullable=False),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.Enum('open', 'active', 'pending_review', 'closed', 'archived',
                                     name='casestatus'), nullable=True),
        sa.Column('client_name', sa.String(200), nullable=True),
        sa.Column('attorney_name', sa.String(200), nullable=True),
        sa.Column('jurisdiction', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('closed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('case_number')
    )
    op.create_index('ix_cases_case_number', 'cases', ['case_number'])
    op.create_index('ix_cases_status_created', 'cases', ['status', 'created_at'])

    # Create evidence table
    op.create_table(
        'evidence',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('case_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('evidence_type', sa.Enum('text', 'image', 'audio', 'video', 'document', 'email',
                                           name='evidencetypedb'), nullable=False),
        sa.Column('original_filename', sa.String(500), nullable=True),
        sa.Column('mime_type', sa.String(100), nullable=True),
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=True),
        sa.Column('original_hash', sa.String(64), nullable=False),
        sa.Column('storage_path', sa.String(1000), nullable=True),
        sa.Column('status', sa.Enum('pending', 'processing', 'analyzed', 'verified', 'flagged', 'error',
                                    name='evidencestatus'), nullable=True),
        sa.Column('extracted_text', sa.Text(), nullable=True),
        sa.Column('behavioral_indicators', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('fusion_results', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('ocr_results', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('fused_score', sa.Float(), nullable=True),
        sa.Column('fused_classification', sa.String(50), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('analyzed_at', sa.DateTime(), nullable=True),
        sa.Column('verified_at', sa.DateTime(), nullable=True),
        sa.Column('extra_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['case_id'], ['cases.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_evidence_case_status', 'evidence', ['case_id', 'status'])
    op.create_index('ix_evidence_type_created', 'evidence', ['evidence_type', 'created_at'])
    op.create_index('ix_evidence_hash', 'evidence', ['original_hash'])

    # Create chain_of_custody table
    op.create_table(
        'chain_of_custody',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('evidence_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('agent_id', sa.String(100), nullable=False),
        sa.Column('agent_type', sa.String(50), nullable=False),
        sa.Column('action', sa.String(200), nullable=False),
        sa.Column('input_hash', sa.String(64), nullable=False),
        sa.Column('output_hash', sa.String(64), nullable=False),
        sa.Column('processing_time_ms', sa.Float(), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=True, default=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('signature', sa.String(512), nullable=True),
        sa.Column('signer_id', sa.String(100), nullable=True),
        sa.Column('extra_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['evidence_id'], ['evidence.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_custody_evidence_timestamp', 'chain_of_custody', ['evidence_id', 'timestamp'])

    # Create analysis_results table
    op.create_table(
        'analysis_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('evidence_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('agent_type', sa.String(50), nullable=False),
        sa.Column('agent_id', sa.String(100), nullable=True),
        sa.Column('result_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('processing_time_ms', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['evidence_id'], ['evidence.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_analysis_evidence_agent', 'analysis_results', ['evidence_id', 'agent_type'])

    # Create analysis_jobs table
    op.create_table(
        'analysis_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('evidence_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('status', sa.String(20), nullable=True, default='pending'),
        sa.Column('current_stage', sa.String(50), nullable=True),
        sa.Column('progress_percent', sa.Float(), nullable=True, default=0),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.BigInteger(), nullable=True, default=0),
        sa.ForeignKeyConstraint(['evidence_id'], ['evidence.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_jobs_status', 'analysis_jobs', ['status'])

    # Create audit_log table
    op.create_table(
        'audit_log',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('resource_type', sa.String(50), nullable=True),
        sa.Column('resource_id', sa.String(100), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.String(500), nullable=True),
        sa.Column('old_value', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('new_value', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_audit_timestamp', 'audit_log', ['timestamp'])
    op.create_index('ix_audit_user_action', 'audit_log', ['user_id', 'action'])


def downgrade() -> None:
    op.drop_index('ix_audit_user_action', 'audit_log')
    op.drop_index('ix_audit_timestamp', 'audit_log')
    op.drop_table('audit_log')

    op.drop_index('ix_jobs_status', 'analysis_jobs')
    op.drop_table('analysis_jobs')

    op.drop_index('ix_analysis_evidence_agent', 'analysis_results')
    op.drop_table('analysis_results')

    op.drop_index('ix_custody_evidence_timestamp', 'chain_of_custody')
    op.drop_table('chain_of_custody')

    op.drop_index('ix_evidence_hash', 'evidence')
    op.drop_index('ix_evidence_type_created', 'evidence')
    op.drop_index('ix_evidence_case_status', 'evidence')
    op.drop_table('evidence')

    op.drop_index('ix_cases_status_created', 'cases')
    op.drop_index('ix_cases_case_number', 'cases')
    op.drop_table('cases')

    op.drop_table('users')

    op.execute('DROP TYPE IF EXISTS evidencestatus')
    op.execute('DROP TYPE IF EXISTS evidencetypedb')
    op.execute('DROP TYPE IF EXISTS casestatus')
