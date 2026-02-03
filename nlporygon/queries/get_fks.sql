SELECT DISTINCT
    tc.table_name,
    tc.constraint_catalog AS table_catalog,
    tc.constraint_schema AS table_schema,
    kcu.column_name,
    ccu.table_name AS referenced_table_name,
    ccu.constraint_catalog AS referenced_table_catalog,
    ccu.constraint_schema AS referenced_table_schema,
    ccu.column_name AS referenced_column
FROM information_schema.table_constraints tc
LEFT JOIN information_schema.key_column_usage kcu
  ON tc.constraint_name = kcu.constraint_name
LEFT JOIN information_schema.constraint_column_usage ccu
  ON tc.constraint_name = ccu.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'