# file: create_tables.py

from sqlalchemy import create_engine, MetaData
from main import metadata  # imports the metadata from your main.py

# Updated to use the "mindcare" database instead of "postgres"
DATABASE_URL = "postgresql://postgres:affan@localhost:5432/mindcare"
engine = create_engine(DATABASE_URL)

metadata.create_all(engine)
print("✅ Tables created in 'mindcare' database!")
