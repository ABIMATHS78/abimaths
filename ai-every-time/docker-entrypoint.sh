#!/bin/sh
# Hosts often attach storage owned by root. Started as root, hand /data to the app's own user and
# run the app as that user; started as some other user already, as some hosts do, just run it.
set -e
if [ "$(id -u)" = "0" ]; then
    mkdir -p /data
    chown -R aiet:aiet /data || echo "Couldn't give /data to the app's user; the app may not be able to save." >&2
    exec setpriv --reuid=aiet --regid=aiet --init-groups "$@"
fi
exec "$@"
