
DIR=`pwd`

for dir in $DIR; do
    cd $dir;
    du -sh * 2>&1 | grep --invert-match "Operation not permitted" | gsort -h | awk '{print "|"$2"\t|"$1;}'
    cd -;
done
