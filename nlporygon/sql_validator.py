import re

_FORBIDDEN_PATTERNS: list[tuple[re.Pattern, str]] = [
    # DDL statements
    (re.compile(r'\b(CREATE|ALTER|DROP)\s+(TABLE|INDEX|VIEW|DATABASE|SCHEMA|FUNCTION|PROCEDURE|TRIGGER)\b', re.IGNORECASE), 'DDL statement'),
    (re.compile(r'\bTRUNCATE\s+TABLE\b', re.IGNORECASE), 'TRUNCATE statement'),

    # DML statements (non-SELECT)
    (re.compile(r'\bINSERT\s+INTO\b', re.IGNORECASE), 'INSERT statement'),
    (re.compile(r'\bUPDATE\s+\w+\s+SET\b', re.IGNORECASE), 'UPDATE statement'),
    (re.compile(r'\bDELETE\s+FROM\b', re.IGNORECASE), 'DELETE statement'),
    (re.compile(r'\bMERGE\s+INTO\b', re.IGNORECASE), 'MERGE statement'),

    # Permission/access control
    (re.compile(r'\b(GRANT|REVOKE)\b', re.IGNORECASE), 'Permission statement'),

    # Transaction control
    (re.compile(r'\b(COMMIT|ROLLBACK)\b', re.IGNORECASE), 'Transaction control'),

    # Dangerous functions/commands
    (re.compile(r'\bEXEC(UTE)?\s*\(', re.IGNORECASE), 'EXECUTE statement'),
    (re.compile(r'\b(XP_|SP_|DBMS_|UTL_)\w+', re.IGNORECASE), 'Stored procedure'),
    (re.compile(r'\bLOAD\s+DATA\b', re.IGNORECASE), 'LOAD DATA statement'),
    (re.compile(r'\bINTO\s+(OUTFILE|DUMPFILE)\b', re.IGNORECASE), 'File output clause'),

    # SQL injection patterns
    (re.compile(r';\s*(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE)', re.IGNORECASE), 'Multiple statements'),
    (re.compile(r'--\s*$', re.IGNORECASE), 'SQL comment injection'),
    (re.compile(r'/\*.*\*/', re.IGNORECASE), 'Block comment injection'),
    (re.compile(r'\bUNION\s+ALL\s+SELECT\s+NULL', re.IGNORECASE), 'UNION NULL injection'),
    (re.compile(r"'\s*OR\s+'?1'?\s*=\s*'?1|OR\s+'1'\s*=\s*'1'", re.IGNORECASE), 'OR 1=1 injection'),
    (re.compile(r'\bSLEEP\s*\(\d+\)', re.IGNORECASE), 'SLEEP timing attack'),
    (re.compile(r'\bBENCHMARK\s*\(', re.IGNORECASE), 'BENCHMARK timing attack'),
    (re.compile(r'\bWAITFOR\s+DELAY\b', re.IGNORECASE), 'WAITFOR timing attack'),

    # File system access
    (re.compile(r'\b(READ_FILE|WRITE_FILE|LOAD_FILE)\s*\(', re.IGNORECASE), 'File access function'),
]


class UnsafeSQLError(Exception):
    """Raised when LLM generates SQL that contains forbidden/dangerous patterns."""

    def __init__(self, message: str, pattern_name: str, sql: str):
        self.pattern_name = pattern_name
        self.sql = sql
        super().__init__(message)


def validate_sql(sql: str) -> None:
    """
    Check SQL for dangerous patterns and raise UnsafeSQLError if found.

    Uses pre-compiled regex patterns for efficiency. This is a defense-in-depth
    measure - the LLM should only generate SELECT statements, but this catches
    any attempts at DDL, DML, or injection.
    """
    for pattern, name in _FORBIDDEN_PATTERNS:
        if pattern.search(sql):
            raise UnsafeSQLError(
                f"Unsafe SQL detected: {name}",
                pattern_name=name,
                sql=sql
            )
