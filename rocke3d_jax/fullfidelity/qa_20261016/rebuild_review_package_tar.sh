#!/bin/bash
set -e
# Usage: REVIEW_PKG=<path of the package directory> bash rebuild_review_package_tar.sh   (the tar and its .sha256 are written next to the directory)
PKG=${REVIEW_PKG:?set REVIEW_PKG to the review package directory}
cd "$(dirname "$PKG")"
N=$(basename "$PKG")
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
