#!/bin/bash
set -euo pipefail

echo "--- [1/2] Running initial DB setup as SYSTEM ---"
# Подключаемся к корню (FREE), скрипт сам переключится в FREEPDB1
sqlplus -L -s "system/${ORACLE_PWD}@//localhost:1521/FREE" @/opt/oracle/scripts/custom/01-create-user-and-tablespace.sql
echo "--- Initial DB setup finished successfully ---"
echo ""

echo "--- [2/2] Running application schema script as E3_ADMIN ---"
/opt/oracle/scripts/custom/02-create-tables.sh

echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ">>>   DATABASE SETUP IS COMPLETE!      <<<"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"