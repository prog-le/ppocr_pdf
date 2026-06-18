---
session: ses_1274
updated: 2026-06-18T04:15:32.930Z
---

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="bash">
<｜｜DSML｜｜parameter name="command" string="true">Get-ChildItem -Recurse C:\guole\code\ppocr_pdf\tests\*.py | Select-Object Name</｜｜DSML｜｜parameter>
<｜｜DSML｜｜parameter name="description" string="true">List all test files</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
<｜｜DSML｜｜invoke name="bash">
<｜｜DSML｜｜parameter name="command" string="true">git diff --stat</｜｜DSML｜｜parameter>
<｜｜DSML｜｜parameter name="description" string="true">Check current diff stats</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
</｜｜DSML｜｜tool_calls>
