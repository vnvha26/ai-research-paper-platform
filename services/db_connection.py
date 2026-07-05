import streamlit as st
import psycopg2
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

@st.cache_resource
def get_db_connections():
    """
    Mở kết nối tới 3 Database và Load Model AI (Chỉ chạy 1 lần khi bật Web).
    """
    connections = {}
    try:
        connections["pg"] = psycopg2.connect(
            dbname="paper_recommender", 
            user="postgresql", 
            password="postgresql", 
            host="localhost"
        )
        
        connections["neo4j"] = GraphDatabase.driver(
            "bolt://localhost:7687", 
            auth=("neo4j", "neo4jvha2601")
        )
        
        connections["qdrant"] = QdrantClient(url="http://localhost:6333")
        
        connections["nlp_model"] = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Khởi tạo thành công")
        return connections
        
    except Exception as e:
        pg_conn = connections.get("pg")
        if pg_conn is not None:
            pg_conn.close()
        neo4j_driver = connections.get("neo4j")
        if neo4j_driver is not None:
            neo4j_driver.close()
        st.error(f"Lỗi khởi tạo Database: {e}")
        return None
