# Hotaisle Connection Guide

## Steps

1. SSH to the admin portal:
   ```bash
   ssh admin.hotaisle.app
   ```

2. Manually start an instance and update `~/.ssh/config` with the new IP/hostname

3. Connect to your instance:
   ```bash
   ssh hotaisle
   ```

## Current SSH Config

```
Host hotaisle
    HostName 23.183.40.90
    User hotaisle
```
