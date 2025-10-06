#!/bin/bash
set -euo pipefail

echo "--- [1/3] Running initial DB setup as SYSTEM ---"
# Подключаемся к корню (FREE), скрипт сам переключится в FREEPDB1
sqlplus -L -s "system/${ORACLE_PWD}@//localhost:1521/FREE" @/opt/oracle/scripts/custom/01-create-user-and-tablespace.sql
echo "--- DB setup finished successfully ---"
echo ""

echo "--- [2/3] Running application schema script as E3_ADMIN ---"
sqlplus -L -s "E3_ADMIN/ddbadmine3@//localhost:1521/${ORACLE_PDB}" <<EOF
WHENEVER SQLERROR EXIT SQL.SQLCODE;
@/opt/oracle/scripts/custom/E3_ADMIN_scheme2_fixed.sql
EXIT;
EOF







echo "--- [3/3] Loading CSV data into E3_ADMIN ---"
/opt/oracle/scripts/custom/03-load-csv.sh

echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ">>>   DATABASE SETUP IS COMPLETE!      <<<"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"