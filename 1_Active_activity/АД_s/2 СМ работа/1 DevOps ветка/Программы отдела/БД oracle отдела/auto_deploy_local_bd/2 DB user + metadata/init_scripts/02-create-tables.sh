#!/bin/bash
set -euo pipefail

echo "--- Running application schema script as E3_ADMIN ---"

sqlplus -L -s "E3_ADMIN/ddbadmine3@//localhost:1521/${ORACLE_PDB}" <<EOF
WHENEVER SQLERROR EXIT SQL.SQLCODE;
@/opt/oracle/scripts/custom/E3_ADMIN_scheme2_fixed.sql
EXIT;
EOF

echo "--- Application schema script finished successfully ---"