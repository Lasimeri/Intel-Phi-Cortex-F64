#!/usr/bin/env bash
# card-cat.sh REMOTE LOCAL: copy a card file to the host (the card has no sftp-server, so scp fails); card-cat.sh -p LOCAL REMOTE copies to the card
set -e
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ "$1" = -p ]; then "$here/card-ssh.sh" "cat > '$3'" < "$2"; else "$here/card-ssh.sh" "cat '$1'" > "$2"; fi
