Note:

This code can be run in a VM in simulation mode, however, in that setting I couldn't get the only existing debugger for sgx (sgx-gdb) to work. And you can't really develop anything if you're not able to see what's going on inside the enclave.

Therefore I really recommend running this on a native linux installation

    Check the sgxsdk install guides for corresponding information

Further, once the sdk is installed, I'd first try to run one of the sample enclaves included in sgxsdk.

Once that works, you can use a softlink such that this project here finds the sdk.

At this point you can use the commands below.

---------------------------------------------
How to Build/Execute the C++11 sample program
---------------------------------------------
1. Install Intel(R) Software Guard Extensions (Intel(R) SGX) SDK for Linux* OS
2. Make sure your environment is set:
    $ source ${sgx-sdk-install-path}/environment
3. Build the project with the prepared Makefile:
    a. Hardware Mode, Debug build:
        $ make
    b. Hardware Mode, Pre-release build:
        $ make SGX_PRERELEASE=1 SGX_DEBUG=0
    c. Hardware Mode, Release build:
        $ make SGX_DEBUG=0
    d. Simulation Mode, Debug build:
        $ make SGX_MODE=SIM
    e. Simulation Mode, Pre-release build:
        $ make SGX_MODE=SIM SGX_PRERELEASE=1 SGX_DEBUG=0
    f. Simulation Mode, Release build:
        $ make SGX_MODE=SIM SGX_DEBUG=0
4. Execute the binary directly:
    $ ./app
5. Remember to "make clean" before switching build mode
