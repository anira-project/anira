# targets
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7a)

# compiler settings
set(CMAKE_C_COMPILER /usr/bin/arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabihf-g++)

# give an option to distinguish
set(BELA TRUE)

set(CMAKE_SYSROOT /sysroot)
set(BELA_ROOT "${CMAKE_SYSROOT}/root/Bela")