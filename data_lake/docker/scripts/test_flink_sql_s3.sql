CREATE TABLE test_users_s3 (
  id INT,
  name STRING,
  age INT,
  city STRING
) WITH (
  'connector' = 'filesystem',
  'path' = 's3://warehouse/test_users_s3/',
  'format' = 'csv'
);

SET 'execution.checkpointing.interval' = '10s';
INSERT INTO test_users_s3 VALUES
  (1, 'Alice', 30, 'Beijing'),
  (2, 'Bob', 25, 'Shanghai'),
  (3, 'Charlie', 35, 'Guangzhou'),
  (4, 'Diana', 28, 'Shenzhen');

SET 'sql-client.execution.result-mode' = 'tableau';
SELECT * FROM test_users_s3;
