#How to build?

1. Make sure what your platform, dose have compiler.
1. Check files in Makefiles directory.
1. Determine what you need to build target.
1. Refer to following cases.

## Dynamic library
1. Building dynamic library should able with this,
	`$ make -f Makefiles/Makefie.windll`
	or for Mac,
	`$ make -f Makefiles/Makefile.macos`
	or for Linux,
	`$ make -f Makefiles/Makefile.linux`

## Static library
1. Appens 'static' at each platform target makefile.
    eg.
	    `$ make -f Makefiles/Makefile.macos static`

## Big Sur with Apple Silicon (M1)
1. Makefile sens arm64 with uname -m on MacOS, and generates universal binary for Intel and Silicon.

## testing unit
1. Testing program will be availed only for console/shell base.
1. It should be able with do this,
	`$ make -f Makefiles/Makefile.test`
1. Some other testing availed for these Makefiles:
	- Makefile.testwin
	- Makefile.testmac
