
DIR='/Users/nextchen'

CUR_DIR=`pwd`
for dir in $DIR; do
    sudo du -sh $dir"/*" 2>&1 | grep --invert-match "Operation not permitted" | gsort -h # | awk '{print "'"$dir"',"$2","$1;}' #> "$CUR_DIR/csv"
    # du -sh "$dir/.*" 2>&1 | grep --invert-match "Operation not permitted" | gsort -h | awk '{print $2"."$1;}'
done
