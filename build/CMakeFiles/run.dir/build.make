# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/arvr/Desktop/ViewAndTime

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/arvr/Desktop/ViewAndTime/build

# Include any dependencies generated for this target.
include CMakeFiles/run.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/run.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/run.dir/flags.make

CMakeFiles/run.dir/src/main.cpp.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/arvr/Desktop/ViewAndTime/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/run.dir/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run.dir/src/main.cpp.o -c /home/arvr/Desktop/ViewAndTime/src/main.cpp

CMakeFiles/run.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run.dir/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/arvr/Desktop/ViewAndTime/src/main.cpp > CMakeFiles/run.dir/src/main.cpp.i

CMakeFiles/run.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run.dir/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/arvr/Desktop/ViewAndTime/src/main.cpp -o CMakeFiles/run.dir/src/main.cpp.s

CMakeFiles/run.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/run.dir/src/main.cpp.o.requires

CMakeFiles/run.dir/src/main.cpp.o.provides: CMakeFiles/run.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/run.dir/src/main.cpp.o.provides

CMakeFiles/run.dir/src/main.cpp.o.provides.build: CMakeFiles/run.dir/src/main.cpp.o


# Object files for target run
run_OBJECTS = \
"CMakeFiles/run.dir/src/main.cpp.o"

# External object files for target run
run_EXTERNAL_OBJECTS =

../run: CMakeFiles/run.dir/src/main.cpp.o
../run: CMakeFiles/run.dir/build.make
../run: ../lib/libSLIC.so
../run: /usr/local/lib/libopencv_videostab.so.2.4.13
../run: /usr/local/lib/libopencv_ts.a
../run: /usr/local/lib/libopencv_superres.so.2.4.13
../run: /usr/local/lib/libopencv_stitching.so.2.4.13
../run: /usr/local/lib/libopencv_contrib.so.2.4.13
../run: /usr/local/lib/libopencv_nonfree.so.2.4.13
../run: /usr/local/lib/libopencv_ocl.so.2.4.13
../run: /usr/local/lib/libopencv_gpu.so.2.4.13
../run: /usr/local/lib/libopencv_photo.so.2.4.13
../run: /usr/local/lib/libopencv_objdetect.so.2.4.13
../run: /usr/local/lib/libopencv_legacy.so.2.4.13
../run: /usr/local/lib/libopencv_video.so.2.4.13
../run: /usr/local/lib/libopencv_ml.so.2.4.13
../run: /usr/local/cuda/lib64/libcufft.so
../run: /usr/local/lib/libopencv_calib3d.so.2.4.13
../run: /usr/local/lib/libopencv_features2d.so.2.4.13
../run: /usr/local/lib/libopencv_highgui.so.2.4.13
../run: /usr/local/lib/libopencv_imgproc.so.2.4.13
../run: /usr/local/lib/libopencv_flann.so.2.4.13
../run: /usr/local/lib/libopencv_core.so.2.4.13
../run: /usr/local/cuda/lib64/libcudart.so
../run: /usr/local/cuda/lib64/libnppc.so
../run: /usr/local/cuda/lib64/libnppi.so
../run: /usr/local/cuda/lib64/libnpps.so
../run: CMakeFiles/run.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/arvr/Desktop/ViewAndTime/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../run"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/run.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/run.dir/build: ../run

.PHONY : CMakeFiles/run.dir/build

CMakeFiles/run.dir/requires: CMakeFiles/run.dir/src/main.cpp.o.requires

.PHONY : CMakeFiles/run.dir/requires

CMakeFiles/run.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/run.dir/cmake_clean.cmake
.PHONY : CMakeFiles/run.dir/clean

CMakeFiles/run.dir/depend:
	cd /home/arvr/Desktop/ViewAndTime/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/arvr/Desktop/ViewAndTime /home/arvr/Desktop/ViewAndTime /home/arvr/Desktop/ViewAndTime/build /home/arvr/Desktop/ViewAndTime/build /home/arvr/Desktop/ViewAndTime/build/CMakeFiles/run.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/run.dir/depend

