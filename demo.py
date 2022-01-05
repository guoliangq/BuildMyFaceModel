import pymysql
import time
db = pymysql.connect(host="localhost", user="root", password="zxd990501", database="recordsystem")
cursor = db.cursor()
createdate = time.strftime("%Y-%m-%d", time.localtime())
print(createdate)
cursor.execute(f"update record set success=0 where account = 20175119 and createdate='{createdate}'")
db.commit()