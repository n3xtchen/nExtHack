
USE CATALOG default_catalog;


DROP TABLE IF EXISTS hudi_taxis;

CREATE TABLE hudi_taxis (
    vendor_id BIGINT,
    trip_id BIGINT,
    trip_distance FLOAT,
    fare_amount DOUBLE,
    store_and_fwd_flag STRING
) WITH (
    'connector' = 'hudi',
    'path' = 's3a://warehouse/hudi/taxi',
    'table.type' = 'COPY_ON_WRITE',
		'hoodie.datasource.write.recordkey.field' = 'trip_id',
    'write.metadata.enabled' = 'false',
    'hoodie.metadata.enable' = 'false',
    'hoodie.write.concurrency.mode' = 'single_writer',
    'hoodie.cleaner.policy.failed.writes' = 'LAZY',
    'hoodie.write.lock.provider' = 'org.apache.hudi.client.transaction.lock.InProcessLockProvider'
);


SET 'execution.checkpointing.interval' = '10s';

INSERT INTO hudi_taxis
VALUES
   (1, 1000371, 1.8, 15.32, 'N'),
   (2, 1000372, 2.5, 22.15, 'N'),
   (2, 1000373, 0.9, 9.01, 'N'),
   (1, 1000374, 8.4, 42.13, 'Y');
 
SET 'sql-client.execution.result-mode' = 'tableau';
SELECT * FROM hudi_taxis;
