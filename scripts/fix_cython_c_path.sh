#!/bin/bash

srcdir=$1
bindir=$2

fix_file() {
    local file=$1
    local timestamp=$(stat -c %y "$file")

    while IFS= read line; do
        if ! [[ $line =~ ^( *)/\*\ \"(.*)\.pxd\":([0-9]+)$ ]]; then
            printf '%s\n' "$line"
            continue
        fi
        prefix=${BASH_REMATCH[1]}
        basename=${BASH_REMATCH[2]}
        lineno=${BASH_REMATCH[3]}
        if [[ -f "$srcdir/brassboard_seq/${basename}.pxd" ]]; then
            printf '%s/* "%s":%s\n' "$prefix" "brassboard_seq/${basename}.pxd" "$lineno"
            continue
        fi
        if [[ -f "$srcdir/tests/${basename}.pxd" ]]; then
            printf '%s/* "%s":%s\n' "$prefix" "tests/${basename}.pxd" "$lineno"
            continue
        fi
        if [[ -f "$srcdir/${basename}.pxd" ]]; then
            printf '%s\n' "$line"
            continue
        fi
        # If we can't find the pxd file, delete it from the comment
        # since it'll otherwise mess up the source file mapping...
        printf '%s/*\n' "$prefix"
    done < "$file" > "$file".tmp

    mv "$file".tmp "$file"

    touch -c -d "$timestamp" "$file"
}

for f in "$bindir/brassboard_seq/"*.cpp "$bindir/tests/"*.cpp; do
    fix_file "$f"
done
