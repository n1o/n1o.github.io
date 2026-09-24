{ pkgs, lib, config, ... }:

let
  # Runtime shared libraries for headless Chromium (omp-managed Chrome,
  # Playwright downloads, etc.) — this WSL/NixOS image lacks glib/nss/gtk.
  # NOTE: scoped to the `screenshot` script, NOT exported globally: mixing
  # these (glibc 2.42 closure) libs into system binaries built against
  # glibc 2.40 (like the devenv CLI itself) crashes them with
  # GLIBC_ABI_GNU2_TLS errors.
  chromiumLibs = lib.makeLibraryPath [
    pkgs.glib
    pkgs.nss
    pkgs.nspr
    pkgs.atk
    pkgs.at-spi2-atk
    pkgs.cups
    pkgs.dbus
    pkgs.expat
    pkgs.libdrm
    pkgs.libxkbcommon
    pkgs.mesa
    pkgs.libgbm
    pkgs.pango
    pkgs.cairo
    pkgs.alsa-lib
    pkgs.xorg.libX11
    pkgs.xorg.libXcomposite
    pkgs.xorg.libXdamage
    pkgs.xorg.libXext
    pkgs.xorg.libXfixes
    pkgs.xorg.libXrandr
    pkgs.xorg.libxcb
    pkgs.fontconfig
    pkgs.freetype
    pkgs.gdk-pixbuf
    pkgs.gtk3
  ];
in
{
  packages = [
    pkgs.nodejs
  ];

  languages.javascript = {
    enable = true;
    npm = {
      enable = true;
      install.enable = true;
    };
  };

  # Port 4321 is permanently taken by the codebreakers frontend dev server;
  # pin this site to 8081 so URLs are stable.
  processes.site.exec = "npm run dev -- --host --port 8081";

  scripts = {
    dev.exec = "npm run dev -- --host --port 8081";
    build.exec = "npm run build";
    preview.exec = "npm run preview -- --host --port 8081";

    # Screenshot a page of the locally previewed build with headless Chromium.
    screenshot.exec = ''
      case "''${1:-}" in -h|--help)
        echo "Usage: screenshot <path> [outfile]"
        echo "  Render http://localhost:8081<path> to a PNG with headless Chromium."
        exit 0 ;; esac
      path="''${1:-/}"
      out="''${2:-screenshot.png}"
      for chrome in "$HOME/.omp/puppeteer/chrome"/linux-*/chrome-linux64/chrome \
                    "$HOME/.cache/ms-playwright"/chromium-*/chrome-linux64/chrome; do
        [ -f "$chrome" ] && break
      done
      [ -f "''${chrome:-}" ] || { echo "No Chromium binary found"; exit 1; }
      LD_LIBRARY_PATH="${chromiumLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$chrome" --headless --no-sandbox --disable-gpu \
        --window-size=1280,1600 --screenshot="$out" "http://localhost:8081$path"
      echo "Wrote $out"
    '';
  };

  enterShell = ''
    echo "n1o.github.io - Astro dev environment"
    echo ""
    echo "Dev:       dev  (astro dev server at http://localhost:8081)"
    echo "Build:     build  preview  screenshot <path>"
    echo "Up:        devenv up  (runs the dev server as a process)"
  '';
}
