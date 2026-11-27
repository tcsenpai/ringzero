## ADDED Requirements

### Requirement: VNC_PASS
`make run-remote` SHALL protect its VNC framebuffer with the password VNC_PASS, default `ringzero`.
Source: Makefile VNC_PASS, target run-remote

#### Scenario: Default password
- **WHEN** the user runs `make run-remote` without VNC_PASS and connects a VNC client to port 5901
- **THEN** the client asks for a password and `ringzero` opens the framebuffer
