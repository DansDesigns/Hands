from types import SimpleNamespace

import install_d1_model


def test_choose_server_binary_skips_tiny_stub_and_uses_real_binary():
    stub = SimpleNamespace(filename="bin/llama-server.exe", file_size=9216)
    binary = SimpleNamespace(filename="llama-server.exe", file_size=32_000_000)
    assert install_d1_model._choose_server_executable([stub, binary]) is binary


def test_choose_server_binary_returns_none_when_only_tiny_entries_exist():
    stub = SimpleNamespace(filename="llama-server.exe", file_size=10752)
    assert install_d1_model._choose_server_executable([stub]) is None


def test_choose_server_binary_picks_largest_valid_match():
    smaller = SimpleNamespace(filename="a/llama-server.exe", file_size=120_000)
    larger = SimpleNamespace(filename="b/llama-server.exe", file_size=28_000_000)
    assert install_d1_model._choose_server_executable([smaller, larger]) is larger


def test_small_launcher_is_accepted_when_server_impl_library_is_beside_it():
    # Current llama.cpp releases ship a small llama-server launcher plus
    # llama-server-impl.dll / libllama-server-impl.so holding the real server.
    launcher = SimpleNamespace(filename="pkg/llama-server.exe", file_size=9216)
    assert install_d1_model._choose_server_executable([launcher]) is None
    assert install_d1_model._choose_server_executable([launcher], launcher_dirs=["pkg"]) is launcher


def _release(tag, names):
    return {"tag_name": tag, "assets": [{"name": n, "browser_download_url": "https://example/" + n} for n in names]}


RELEASE = _release("b7000", [
    "llama-b7000-bin-win-cpu-x64.zip", "llama-b7000-bin-win-cpu-arm64.zip",
    "llama-b7000-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip",
    "llama-b7000-bin-win-vulkan-x64.zip", "llama-b7000-bin-win-opencl-adreno-arm64.zip",
    "llama-b7000-bin-ubuntu-x64.tar.gz", "llama-b7000-bin-ubuntu-arm64.tar.gz",
    "llama-b7000-bin-ubuntu-s390x.tar.gz", "llama-b7000-bin-ubuntu-vulkan-x64.tar.gz",
    "llama-b7000-bin-linux-arm64-snapdragon.tar.gz", "llama-b7000-bin-android-arm64.tar.gz",
    "llama-b7000-bin-macos-arm64.tar.gz", "llama-b7000-ui.tar.gz",
])


def _best(os_tag, arch):
    found = install_d1_model.select_release_assets([RELEASE], os_tag, arch)
    return found[0][1] if found else None


def test_cpu_package_is_selected_for_each_windows_and_linux_architecture():
    assert _best("windows", "x64") == "llama-b7000-bin-win-cpu-x64.zip"
    assert _best("windows", "arm64") == "llama-b7000-bin-win-cpu-arm64.zip"
    assert _best("linux", "x64") == "llama-b7000-bin-ubuntu-x64.tar.gz"
    assert _best("linux", "arm64") == "llama-b7000-bin-ubuntu-arm64.tar.gz"
    assert _best("linux", "s390x") == "llama-b7000-bin-ubuntu-s390x.tar.gz"


def test_architectures_without_prebuilt_packages_fall_back_to_source_build():
    for arch in ("riscv64", "arm32", "ppc64le"):
        assert _best("linux", arch) is None
