#!/bin/zsh
set -eu

read -rs "kuailab_key?Paste the OpenAI API key (input hidden), then press Enter: "
print
export OPENAI_API_KEY="$kuailab_key"
unset kuailab_key
exec npm run local
