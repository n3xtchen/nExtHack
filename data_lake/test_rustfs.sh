
export AWS_ACCESS_KEY_ID=n3xtchen
export AWS_SECRET_ACCESS_KEY=n3xtchen
export AWS_ENDPOINT_URL=http://localhost:9000

aws s3 mb s3://test
aws s3 ls 
aws s3 cp start_rustfs.sh s3://test/
aws s3 ls s3://test/
aws s3 cp s3://test/start_rustfs.sh test.sh
rm test.sh
aws s3 rm  s3://test/start_rustfs.sh
aws s3 ls s3://test/
aws s3 rb  s3://test/
