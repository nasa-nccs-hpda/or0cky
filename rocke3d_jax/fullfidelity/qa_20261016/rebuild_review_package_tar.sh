#!/bin/bash
set -e
cd /panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai
N=rocke3d_review_package_20261016
echo "manifest check..."
(cd $N && sha256sum -c MANIFEST.sha256 --quiet) && echo "MANIFEST_OK"
rm -f $N.tar.new
tar -cf $N.tar.new $N
tar -tf $N.tar.new > /dev/null
H=$(sha256sum $N.tar.new | cut -d' ' -f1)
command mv -f $N.tar.new $N.tar
echo "$H  $N.tar" > $N.tar.sha256
sha256sum -c $N.tar.sha256
echo "amended criteria lines in tar: $(tar -xOf $N.tar $N/ACCEPTANCE_CRITERIA.md | grep -c 'Amended 2026-10-16')"
echo "plot files in tar: $(tar -tf $N.tar | grep -c "$N/plots/")"
echo REBUILD_DONE
