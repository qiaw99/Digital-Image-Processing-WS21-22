# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/qiaw99/Downloads/dip04

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/qiaw99/Downloads/dip04/build

# Include any dependencies generated for this target.
include CMakeFiles/unit_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/unit_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/unit_test.dir/flags.make

CMakeFiles/unit_test.dir/unit_test.cpp.o: CMakeFiles/unit_test.dir/flags.make
CMakeFiles/unit_test.dir/unit_test.cpp.o: ../unit_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qiaw99/Downloads/dip04/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/unit_test.dir/unit_test.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/unit_test.dir/unit_test.cpp.o -c /home/qiaw99/Downloads/dip04/unit_test.cpp

CMakeFiles/unit_test.dir/unit_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/unit_test.dir/unit_test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qiaw99/Downloads/dip04/unit_test.cpp > CMakeFiles/unit_test.dir/unit_test.cpp.i

CMakeFiles/unit_test.dir/unit_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/unit_test.dir/unit_test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qiaw99/Downloads/dip04/unit_test.cpp -o CMakeFiles/unit_test.dir/unit_test.cpp.s

CMakeFiles/unit_test.dir/unit_test.cpp.o.requires:

.PHONY : CMakeFiles/unit_test.dir/unit_test.cpp.o.requires

CMakeFiles/unit_test.dir/unit_test.cpp.o.provides: CMakeFiles/unit_test.dir/unit_test.cpp.o.requires
	$(MAKE) -f CMakeFiles/unit_test.dir/build.make CMakeFiles/unit_test.dir/unit_test.cpp.o.provides.build
.PHONY : CMakeFiles/unit_test.dir/unit_test.cpp.o.provides

CMakeFiles/unit_test.dir/unit_test.cpp.o.provides.build: CMakeFiles/unit_test.dir/unit_test.cpp.o


# Object files for target unit_test
unit_test_OBJECTS = \
"CMakeFiles/unit_test.dir/unit_test.cpp.o"

# External object files for target unit_test
unit_test_EXTERNAL_OBJECTS =

unit_test: CMakeFiles/unit_test.dir/unit_test.cpp.o
unit_test: CMakeFiles/unit_test.dir/build.make
unit_test: libcode.a
unit_test: /usr/local/lib/libopencv_gapi.so.4.5.4
unit_test: /usr/local/lib/libopencv_stitching.so.4.5.4
unit_test: /usr/local/lib/libopencv_aruco.so.4.5.4
unit_test: /usr/local/lib/libopencv_barcode.so.4.5.4
unit_test: /usr/local/lib/libopencv_bgsegm.so.4.5.4
unit_test: /usr/local/lib/libopencv_bioinspired.so.4.5.4
unit_test: /usr/local/lib/libopencv_ccalib.so.4.5.4
unit_test: /usr/local/lib/libopencv_dnn_objdetect.so.4.5.4
unit_test: /usr/local/lib/libopencv_dnn_superres.so.4.5.4
unit_test: /usr/local/lib/libopencv_dpm.so.4.5.4
unit_test: /usr/local/lib/libopencv_face.so.4.5.4
unit_test: /usr/local/lib/libopencv_freetype.so.4.5.4
unit_test: /usr/local/lib/libopencv_fuzzy.so.4.5.4
unit_test: /usr/local/lib/libopencv_hfs.so.4.5.4
unit_test: /usr/local/lib/libopencv_img_hash.so.4.5.4
unit_test: /usr/local/lib/libopencv_intensity_transform.so.4.5.4
unit_test: /usr/local/lib/libopencv_line_descriptor.so.4.5.4
unit_test: /usr/local/lib/libopencv_mcc.so.4.5.4
unit_test: /usr/local/lib/libopencv_quality.so.4.5.4
unit_test: /usr/local/lib/libopencv_rapid.so.4.5.4
unit_test: /usr/local/lib/libopencv_reg.so.4.5.4
unit_test: /usr/local/lib/libopencv_rgbd.so.4.5.4
unit_test: /usr/local/lib/libopencv_saliency.so.4.5.4
unit_test: /usr/local/lib/libopencv_stereo.so.4.5.4
unit_test: /usr/local/lib/libopencv_structured_light.so.4.5.4
unit_test: /usr/local/lib/libopencv_phase_unwrapping.so.4.5.4
unit_test: /usr/local/lib/libopencv_superres.so.4.5.4
unit_test: /usr/local/lib/libopencv_optflow.so.4.5.4
unit_test: /usr/local/lib/libopencv_surface_matching.so.4.5.4
unit_test: /usr/local/lib/libopencv_tracking.so.4.5.4
unit_test: /usr/local/lib/libopencv_highgui.so.4.5.4
unit_test: /usr/local/lib/libopencv_datasets.so.4.5.4
unit_test: /usr/local/lib/libopencv_plot.so.4.5.4
unit_test: /usr/local/lib/libopencv_text.so.4.5.4
unit_test: /usr/local/lib/libopencv_videostab.so.4.5.4
unit_test: /usr/local/lib/libopencv_videoio.so.4.5.4
unit_test: /usr/local/lib/libopencv_wechat_qrcode.so.4.5.4
unit_test: /usr/local/lib/libopencv_xfeatures2d.so.4.5.4
unit_test: /usr/local/lib/libopencv_ml.so.4.5.4
unit_test: /usr/local/lib/libopencv_shape.so.4.5.4
unit_test: /usr/local/lib/libopencv_ximgproc.so.4.5.4
unit_test: /usr/local/lib/libopencv_video.so.4.5.4
unit_test: /usr/local/lib/libopencv_xobjdetect.so.4.5.4
unit_test: /usr/local/lib/libopencv_imgcodecs.so.4.5.4
unit_test: /usr/local/lib/libopencv_objdetect.so.4.5.4
unit_test: /usr/local/lib/libopencv_calib3d.so.4.5.4
unit_test: /usr/local/lib/libopencv_dnn.so.4.5.4
unit_test: /usr/local/lib/libopencv_features2d.so.4.5.4
unit_test: /usr/local/lib/libopencv_flann.so.4.5.4
unit_test: /usr/local/lib/libopencv_xphoto.so.4.5.4
unit_test: /usr/local/lib/libopencv_photo.so.4.5.4
unit_test: /usr/local/lib/libopencv_imgproc.so.4.5.4
unit_test: /usr/local/lib/libopencv_core.so.4.5.4
unit_test: CMakeFiles/unit_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/qiaw99/Downloads/dip04/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable unit_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/unit_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/unit_test.dir/build: unit_test

.PHONY : CMakeFiles/unit_test.dir/build

CMakeFiles/unit_test.dir/requires: CMakeFiles/unit_test.dir/unit_test.cpp.o.requires

.PHONY : CMakeFiles/unit_test.dir/requires

CMakeFiles/unit_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/unit_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/unit_test.dir/clean

CMakeFiles/unit_test.dir/depend:
	cd /home/qiaw99/Downloads/dip04/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/qiaw99/Downloads/dip04 /home/qiaw99/Downloads/dip04 /home/qiaw99/Downloads/dip04/build /home/qiaw99/Downloads/dip04/build /home/qiaw99/Downloads/dip04/build/CMakeFiles/unit_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/unit_test.dir/depend

