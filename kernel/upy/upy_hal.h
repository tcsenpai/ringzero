// mphalport.h for ringzero (instead of ports/embed's): no pins, Ctrl-C is polled by upy.rs, not by a char hook.
#define mp_hal_pin_obj_t
static inline void mp_hal_set_interrupt_char(int c) {
    (void)c;
}
