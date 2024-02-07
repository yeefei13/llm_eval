#!/bin/bash

download_file() {
    local url=$1
    local output=$2

    echo "Downloading ${output}..."
    wget -O ${output} -c ${url}
    echo "Downloaded ${output}"
}

download_file "https://www.dropbox.com/scl/fi/27kfkzuclnwghbvd4wb8r/validation_queries.tsv?rlkey=h3ty7ix690kvohjzm19to55ch&dl=0" "validation_queries_1.tsv"
download_file "https://www.dropbox.com/scl/fi/4dxuk9ylm1rxddeka05ff/wikipedia_documents.tsv?rlkey=6w8labzzsnlz214up42vsyac7&dl=0" "validation_documents_1.tsv"