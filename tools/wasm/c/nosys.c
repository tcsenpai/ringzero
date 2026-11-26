// M8: libc calls wasi-libc does not have, for the C programs in /bin (sqlite3's .system, lua's os.execute and
// os.tmpname). No processes on ringzero: they fail the way a libc without a shell does.
#include <errno.h>
#include <stdio.h>

int system(const char *cmd) {
    if (!cmd) return 0; // "is there a shell?" no
    errno = ENOSYS;
    return -1;
}

char *tmpnam(char *s) {
    (void)s;
    errno = ENOSYS;
    return NULL;
}

FILE *tmpfile(void) {
    errno = ENOSYS;
    return NULL;
}
