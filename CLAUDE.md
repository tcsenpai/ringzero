# ringzero

Project rules for agents are in AGENTS.md (read it first). The one design rule to keep in mind on every change:

**ringzero presents itself as a Unix.** The models already know Linux, so the OS follows Linux/Unix conventions
wherever they fit: everything is a file (/proc, /sys as sysfs, /dev, /etc), familiar paths, command names and
outputs, POSIX-like behaviour. Expose new state as files first; add a special tool only when a file cannot do
the job. ringzero's own data stays off Linux paths (/var/lib/ringzero).
