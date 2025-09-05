-- Immutable Audit Log System for MPS Connect
-- This file creates database triggers and functions to ensure audit logs cannot be modified

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create immutable audit log table with additional security
CREATE TABLE IF NOT EXISTS immutable_audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(100) NOT NULL,
    record_id UUID NOT NULL,
    action VARCHAR(20) NOT NULL CHECK (action IN ('INSERT', 'UPDATE', 'DELETE')),
    old_values JSONB,
    new_values JSONB,
    user_id UUID REFERENCES users(id),
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- Immutable fields
    hash_chain VARCHAR(64) NOT NULL, -- SHA-256 hash of previous record + current record
    previous_hash VARCHAR(64), -- Hash of previous record
    block_number BIGSERIAL, -- Sequential block number
    is_immutable BOOLEAN DEFAULT true,
    -- Constraints to prevent modification
    CONSTRAINT immutable_audit_logs_immutable CHECK (is_immutable = true)
);

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_immutable_audit_logs_table_record ON immutable_audit_logs(table_name, record_id);
CREATE INDEX IF NOT EXISTS idx_immutable_audit_logs_created_at ON immutable_audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_immutable_audit_logs_hash_chain ON immutable_audit_logs(hash_chain);
CREATE INDEX IF NOT EXISTS idx_immutable_audit_logs_block_number ON immutable_audit_logs(block_number);

-- Function to calculate hash chain
CREATE OR REPLACE FUNCTION calculate_audit_hash(
    p_previous_hash VARCHAR(64),
    p_table_name VARCHAR(100),
    p_record_id UUID,
    p_action VARCHAR(20),
    p_old_values JSONB,
    p_new_values JSONB,
    p_user_id UUID,
    p_ip_address INET,
    p_user_agent TEXT,
    p_created_at TIMESTAMP WITH TIME ZONE
) RETURNS VARCHAR(64) AS $$
DECLARE
    hash_input TEXT;
    calculated_hash VARCHAR(64);
BEGIN
    -- Create hash input string
    hash_input := COALESCE(p_previous_hash, '') || '|' ||
                  p_table_name || '|' ||
                  p_record_id::TEXT || '|' ||
                  p_action || '|' ||
                  COALESCE(p_old_values::TEXT, '') || '|' ||
                  COALESCE(p_new_values::TEXT, '') || '|' ||
                  COALESCE(p_user_id::TEXT, '') || '|' ||
                  COALESCE(p_ip_address::TEXT, '') || '|' ||
                  COALESCE(p_user_agent, '') || '|' ||
                  p_created_at::TEXT;
    
    -- Calculate SHA-256 hash
    calculated_hash := encode(digest(hash_input, 'sha256'), 'hex');
    
    RETURN calculated_hash;
END;
$$ LANGUAGE plpgsql;

-- Function to get previous hash
CREATE OR REPLACE FUNCTION get_previous_audit_hash() RETURNS VARCHAR(64) AS $$
DECLARE
    prev_hash VARCHAR(64);
BEGIN
    SELECT hash_chain INTO prev_hash
    FROM immutable_audit_logs
    ORDER BY block_number DESC
    LIMIT 1;
    
    RETURN COALESCE(prev_hash, '');
END;
$$ LANGUAGE plpgsql;

-- Trigger function to create immutable audit log
CREATE OR REPLACE FUNCTION create_immutable_audit_log() RETURNS TRIGGER AS $$
DECLARE
    prev_hash VARCHAR(64);
    new_hash VARCHAR(64);
    audit_record RECORD;
BEGIN
    -- Get previous hash
    prev_hash := get_previous_audit_hash();
    
    -- Determine action and values
    IF TG_OP = 'INSERT' THEN
        audit_record := ROW(
            TG_TABLE_NAME,
            NEW.id,
            'INSERT',
            NULL,
            to_jsonb(NEW),
            current_setting('app.current_user_id', true)::UUID,
            inet_client_addr(),
            current_setting('app.current_user_agent', true),
            NOW()
        );
    ELSIF TG_OP = 'UPDATE' THEN
        audit_record := ROW(
            TG_TABLE_NAME,
            NEW.id,
            'UPDATE',
            to_jsonb(OLD),
            to_jsonb(NEW),
            current_setting('app.current_user_id', true)::UUID,
            inet_client_addr(),
            current_setting('app.current_user_agent', true),
            NOW()
        );
    ELSIF TG_OP = 'DELETE' THEN
        audit_record := ROW(
            TG_TABLE_NAME,
            OLD.id,
            'DELETE',
            to_jsonb(OLD),
            NULL,
            current_setting('app.current_user_id', true)::UUID,
            inet_client_addr(),
            current_setting('app.current_user_agent', true),
            NOW()
        );
    END IF;
    
    -- Calculate hash
    new_hash := calculate_audit_hash(
        prev_hash,
        audit_record.table_name,
        audit_record.record_id,
        audit_record.action,
        audit_record.old_values,
        audit_record.new_values,
        audit_record.user_id,
        audit_record.ip_address,
        audit_record.user_agent,
        audit_record.created_at
    );
    
    -- Insert immutable audit log
    INSERT INTO immutable_audit_logs (
        table_name,
        record_id,
        action,
        old_values,
        new_values,
        user_id,
        ip_address,
        user_agent,
        created_at,
        hash_chain,
        previous_hash
    ) VALUES (
        audit_record.table_name,
        audit_record.record_id,
        audit_record.action,
        audit_record.old_values,
        audit_record.new_values,
        audit_record.user_id,
        audit_record.ip_address,
        audit_record.user_agent,
        audit_record.created_at,
        new_hash,
        prev_hash
    );
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create triggers for all auditable tables
CREATE TRIGGER immutable_audit_users_trigger
    AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION create_immutable_audit_log();

CREATE TRIGGER immutable_audit_cases_trigger
    AFTER INSERT OR UPDATE OR DELETE ON cases
    FOR EACH ROW EXECUTE FUNCTION create_immutable_audit_log();

CREATE TRIGGER immutable_audit_conversations_trigger
    AFTER INSERT OR UPDATE OR DELETE ON conversations
    FOR EACH ROW EXECUTE FUNCTION create_immutable_audit_log();

CREATE TRIGGER immutable_audit_letters_trigger
    AFTER INSERT OR UPDATE OR DELETE ON letters
    FOR EACH ROW EXECUTE FUNCTION create_immutable_audit_log();

CREATE TRIGGER immutable_audit_sessions_trigger
    AFTER INSERT OR UPDATE OR DELETE ON sessions
    FOR EACH ROW EXECUTE FUNCTION create_immutable_audit_log();

CREATE TRIGGER immutable_audit_permissions_trigger
    AFTER INSERT OR UPDATE OR DELETE ON permissions
    FOR EACH ROW EXECUTE FUNCTION create_immutable_audit_log();

-- Function to verify audit chain integrity
CREATE OR REPLACE FUNCTION verify_audit_chain_integrity() RETURNS TABLE(
    block_number BIGINT,
    table_name VARCHAR(100),
    record_id UUID,
    action VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE,
    hash_chain VARCHAR(64),
    previous_hash VARCHAR(64),
    is_valid BOOLEAN,
    expected_hash VARCHAR(64)
) AS $$
DECLARE
    audit_record RECORD;
    calculated_hash VARCHAR(64);
    prev_hash VARCHAR(64) := '';
    is_valid BOOLEAN;
BEGIN
    FOR audit_record IN 
        SELECT * FROM immutable_audit_logs 
        ORDER BY block_number
    LOOP
        -- Calculate expected hash
        calculated_hash := calculate_audit_hash(
            prev_hash,
            audit_record.table_name,
            audit_record.record_id,
            audit_record.action,
            audit_record.old_values,
            audit_record.new_values,
            audit_record.user_id,
            audit_record.ip_address,
            audit_record.user_agent,
            audit_record.created_at
        );
        
        -- Check if hash matches
        is_valid := (audit_record.hash_chain = calculated_hash);
        
        -- Return record with validation result
        RETURN QUERY SELECT
            audit_record.block_number,
            audit_record.table_name,
            audit_record.record_id,
            audit_record.action,
            audit_record.created_at,
            audit_record.hash_chain,
            audit_record.previous_hash,
            is_valid,
            calculated_hash;
        
        -- Update previous hash for next iteration
        prev_hash := audit_record.hash_chain;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to get audit chain summary
CREATE OR REPLACE FUNCTION get_audit_chain_summary() RETURNS TABLE(
    total_records BIGINT,
    first_record_timestamp TIMESTAMP WITH TIME ZONE,
    last_record_timestamp TIMESTAMP WITH TIME ZONE,
    chain_integrity_status TEXT,
    broken_links_count BIGINT
) AS $$
DECLARE
    total_count BIGINT;
    first_ts TIMESTAMP WITH TIME ZONE;
    last_ts TIMESTAMP WITH TIME ZONE;
    broken_count BIGINT;
    integrity_status TEXT;
BEGIN
    -- Get basic counts
    SELECT COUNT(*), MIN(created_at), MAX(created_at)
    INTO total_count, first_ts, last_ts
    FROM immutable_audit_logs;
    
    -- Count broken links
    SELECT COUNT(*)
    INTO broken_count
    FROM verify_audit_chain_integrity()
    WHERE is_valid = false;
    
    -- Determine integrity status
    IF broken_count = 0 THEN
        integrity_status := 'INTACT';
    ELSIF broken_count < total_count * 0.1 THEN
        integrity_status := 'MINOR_CORRUPTION';
    ELSE
        integrity_status := 'MAJOR_CORRUPTION';
    END IF;
    
    RETURN QUERY SELECT
        total_count,
        first_ts,
        last_ts,
        integrity_status,
        broken_count;
END;
$$ LANGUAGE plpgsql;

-- Function to prevent modification of audit logs
CREATE OR REPLACE FUNCTION prevent_audit_modification() RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Immutable audit logs cannot be modified or deleted. Attempted operation: %', TG_OP;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create triggers to prevent modification
CREATE TRIGGER prevent_immutable_audit_update
    BEFORE UPDATE ON immutable_audit_logs
    FOR EACH ROW EXECUTE FUNCTION prevent_audit_modification();

CREATE TRIGGER prevent_immutable_audit_delete
    BEFORE DELETE ON immutable_audit_logs
    FOR EACH ROW EXECUTE FUNCTION prevent_audit_modification();

-- Function to create audit report
CREATE OR REPLACE FUNCTION generate_immutable_audit_report(
    p_start_date TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    p_end_date TIMESTAMP WITH TIME ZONE DEFAULT NULL
) RETURNS TABLE(
    report_period_start TIMESTAMP WITH TIME ZONE,
    report_period_end TIMESTAMP WITH TIME ZONE,
    total_audit_records BIGINT,
    records_by_action JSONB,
    records_by_table JSONB,
    chain_integrity_status TEXT,
    broken_links_count BIGINT,
    first_audit_timestamp TIMESTAMP WITH TIME ZONE,
    last_audit_timestamp TIMESTAMP WITH TIME ZONE
) AS $$
DECLARE
    start_ts TIMESTAMP WITH TIME ZONE;
    end_ts TIMESTAMP WITH TIME ZONE;
    total_records BIGINT;
    action_breakdown JSONB;
    table_breakdown JSONB;
    integrity_info RECORD;
BEGIN
    -- Set default dates if not provided
    start_ts := COALESCE(p_start_date, (SELECT MIN(created_at) FROM immutable_audit_logs));
    end_ts := COALESCE(p_end_date, (SELECT MAX(created_at) FROM immutable_audit_logs));
    
    -- Get total records in period
    SELECT COUNT(*)
    INTO total_records
    FROM immutable_audit_logs
    WHERE created_at >= start_ts AND created_at <= end_ts;
    
    -- Get action breakdown
    SELECT jsonb_object_agg(action, action_count)
    INTO action_breakdown
    FROM (
        SELECT action, COUNT(*) as action_count
        FROM immutable_audit_logs
        WHERE created_at >= start_ts AND created_at <= end_ts
        GROUP BY action
    ) action_stats;
    
    -- Get table breakdown
    SELECT jsonb_object_agg(table_name, table_count)
    INTO table_breakdown
    FROM (
        SELECT table_name, COUNT(*) as table_count
        FROM immutable_audit_logs
        WHERE created_at >= start_ts AND created_at <= end_ts
        GROUP BY table_name
    ) table_stats;
    
    -- Get integrity information
    SELECT * INTO integrity_info FROM get_audit_chain_summary();
    
    RETURN QUERY SELECT
        start_ts,
        end_ts,
        total_records,
        action_breakdown,
        table_breakdown,
        integrity_info.chain_integrity_status,
        integrity_info.broken_links_count,
        integrity_info.first_record_timestamp,
        integrity_info.last_record_timestamp;
END;
$$ LANGUAGE plpgsql;

-- Grant necessary permissions
GRANT EXECUTE ON FUNCTION calculate_audit_hash TO PUBLIC;
GRANT EXECUTE ON FUNCTION get_previous_audit_hash TO PUBLIC;
GRANT EXECUTE ON FUNCTION create_immutable_audit_log TO PUBLIC;
GRANT EXECUTE ON FUNCTION verify_audit_chain_integrity TO PUBLIC;
GRANT EXECUTE ON FUNCTION get_audit_chain_summary TO PUBLIC;
GRANT EXECUTE ON FUNCTION prevent_audit_modification TO PUBLIC;
GRANT EXECUTE ON FUNCTION generate_immutable_audit_report TO PUBLIC;

-- Create view for easy access to audit logs
CREATE OR REPLACE VIEW audit_logs_view AS
SELECT 
    ial.block_number,
    ial.table_name,
    ial.record_id,
    ial.action,
    ial.old_values,
    ial.new_values,
    u.email as user_email,
    u.name as user_name,
    ial.ip_address,
    ial.user_agent,
    ial.created_at,
    ial.hash_chain,
    ial.previous_hash
FROM immutable_audit_logs ial
LEFT JOIN users u ON ial.user_id = u.id
ORDER BY ial.block_number;

-- Grant read access to audit view
GRANT SELECT ON audit_logs_view TO PUBLIC;
