
CREATE CATALOG iceberg_catalog WITH (
  'type'                 = 'iceberg',
  'catalog-impl'         = 'org.apache.iceberg.rest.RESTCatalog',
  'uri'                  = 'http://iceberg-rest:8181',
  'warehouse'            = 's3://warehouse/iceberg/',
  'io-impl'              = 'org.apache.iceberg.aws.s3.S3FileIO',
  's3.endpoint'          = 'http://rustfs:9000',
  's3.access-key-id'     = 'n3xtchen',
  's3.secret-access-key' = 'n3xtchen',
  's3.path-style-access' = 'true'
);


CREATE DATABASE IF NOT EXISTS iceberg_catalog.nyc;

DROP TABLE IF EXISTS iceberg_catalog.nyc.taxis;

CREATE TABLE iceberg_catalog.nyc.taxis (
    vendor_id BIGINT,
    trip_id BIGINT,
    trip_distance FLOAT,
    fare_amount DOUBLE,
    store_and_fwd_flag STRING
);

SET 'execution.checkpointing.interval' = '10s';

INSERT INTO iceberg_catalog.nyc.taxis
VALUES
   (1, 1000371, 1.8, 15.32, 'N'),
   (2, 1000372, 2.5, 22.15, 'N'),
   (2, 1000373, 0.9, 9.01, 'N'),
   (1, 1000374, 8.4, 42.13, 'Y');
 
SET 'sql-client.execution.result-mode' = 'tableau';
SELECT * FROM iceberg_catalog.nyc.taxis;
