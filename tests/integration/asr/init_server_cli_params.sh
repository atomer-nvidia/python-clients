if [ -z "${SERVER}" ]; then
  server_args="--server localhost:50051"
else
  server_args="--server ${SERVER}"
fi
if [ ! -z "${USE_SSL}" ] && [ "${USE_SSL}" != 0 ]; then
  server_args="${server_args} --use_ssl"
fi
if [ ! -z "${SSL_CERT}" ]; then
  server_args="${server_args} --ssl_cert ${SSL_CERT}"
fi