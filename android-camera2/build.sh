#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
TOOLS_VERSION="${ANDROID_BUILD_TOOLS_VERSION:-36.0.0}"
TOOLS_DIR="$SDK_DIR/build-tools/$TOOLS_VERSION"
ANDROID_JAR="$SDK_DIR/platforms/android-34/android.jar"
BUILD_DIR="$SCRIPT_DIR/build"
CLASSES_DIR="$BUILD_DIR/classes"
DEX_DIR="$BUILD_DIR/dex"
COMPILED_RES_DIR="$BUILD_DIR/compiled-res"

if [[ -z "$SDK_DIR" || ! -f "$ANDROID_JAR" ]]; then
  echo "Set ANDROID_SDK_ROOT to an Android SDK containing platforms;android-34." >&2
  exit 1
fi
for tool in aapt2 aapt d8 zipalign apksigner; do
  if [[ ! -x "$TOOLS_DIR/$tool" ]]; then
    echo "Missing $TOOLS_DIR/$tool. Install build-tools;$TOOLS_VERSION." >&2
    exit 1
  fi
done

mkdir -p "$CLASSES_DIR" "$DEX_DIR" "$COMPILED_RES_DIR"
find "$CLASSES_DIR" "$DEX_DIR" "$COMPILED_RES_DIR" -type f -delete

"$TOOLS_DIR/aapt2" compile --dir "$SCRIPT_DIR/res" -o "$COMPILED_RES_DIR"
mapfile -d '' COMPILED_RESOURCES < <(find "$COMPILED_RES_DIR" -type f -name '*.flat' -print0)

"$TOOLS_DIR/aapt2" link \
  -I "$ANDROID_JAR" \
  --manifest "$SCRIPT_DIR/AndroidManifest.xml" \
  --min-sdk-version 28 \
  --target-sdk-version 34 \
  -o "$BUILD_DIR/base.apk" \
  "${COMPILED_RESOURCES[@]}"

javac --release 17 -classpath "$ANDROID_JAR" -d "$CLASSES_DIR" \
  "$SCRIPT_DIR/src/com/example/pulsecamera2/MainActivity.java"
jar --create --file "$BUILD_DIR/classes.jar" -C "$CLASSES_DIR" .
"$TOOLS_DIR/d8" --lib "$ANDROID_JAR" --min-api 28 --output "$DEX_DIR" "$BUILD_DIR/classes.jar"

cp "$BUILD_DIR/base.apk" "$BUILD_DIR/unsigned.apk"
(cd "$DEX_DIR" && "$TOOLS_DIR/aapt" add "$BUILD_DIR/unsigned.apk" classes.dex >/dev/null)
"$TOOLS_DIR/zipalign" -f 4 "$BUILD_DIR/unsigned.apk" "$BUILD_DIR/aligned.apk"

if [[ ! -f "$BUILD_DIR/debug.keystore" ]]; then
  keytool -genkeypair -v \
    -keystore "$BUILD_DIR/debug.keystore" \
    -storepass android -keypass android \
    -alias androiddebugkey -keyalg RSA -keysize 2048 -validity 10000 \
    -dname "CN=Android Debug,O=Pulse Camera2,C=US" >/dev/null
fi

"$TOOLS_DIR/apksigner" sign \
  --ks "$BUILD_DIR/debug.keystore" \
  --ks-pass pass:android \
  --key-pass pass:android \
  --out "$BUILD_DIR/pulse-camera2-debug.apk" \
  "$BUILD_DIR/aligned.apk"
"$TOOLS_DIR/apksigner" verify "$BUILD_DIR/pulse-camera2-debug.apk"
echo "Built $BUILD_DIR/pulse-camera2-debug.apk"
