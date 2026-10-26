#!/usr/bin/env bash
# card-ssh.sh CMD...: run a command on the 7120A over the ssh forward (works while a long `phi run` holds the control socket)
exec ssh -p 2222 -i "$HOME/.ssh/phi_ed25519" -o BatchMode=yes -o ConnectTimeout=5 -o LogLevel=ERROR -o UserKnownHostsFile="$HOME/.ssh/known_hosts_phi" -o StrictHostKeyChecking=accept-new root@127.0.0.1 "$@"
