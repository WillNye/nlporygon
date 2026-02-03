SELECT DISTINCT
    tc.table_name,
    tc.constraint_catalog AS table_catalog,
    tc.constraint_schema AS table_schema,
    kcu.column_name,
    TRUE AS is_primary_key
FROM information_schema.table_constraints tc
LEFT JOIN information_schema.key_column_usage kcu
  ON tc.constraint_name = kcu.constraint_name
LEFT JOIN information_schema.constraint_column_usage ccu
  ON tc.constraint_name = ccu.constraint_name
WHERE tc.constraint_type = 'PRIMARY KEY'