# Pinned Android toolchain (spec §5): NDK r27c, arm64-v8a.
set(CMAKE_TOOLCHAIN_FILE "/opt/homebrew/share/android-commandlinetools/ndk/27.2.12479018/build/cmake/android.toolchain.cmake" CACHE PATH "")
set(ANDROID_ABI "arm64-v8a" CACHE STRING "")
set(ANDROID_PLATFORM "android-28" CACHE STRING "")
set(ANDROID_STL "c++_static" CACHE STRING "")
