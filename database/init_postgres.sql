CREATE TABLE papers (
    paper_id VARCHAR(50) PRIMARY KEY,
    title TEXT NOT NULL,
    year INT,
    abstract TEXT,
    intro_text TEXT,
    method_text TEXT,
    conclusion_text TEXT,
    authors JSONB,
    search_vector TSVECTOR
);

CREATE FUNCTION papers_search_trigger() RETURNS trigger AS $$
BEGIN
  new.search_vector :=
    setweight(to_tsvector('english', coalesce(new.title,'')), 'A') ||
    setweight(to_tsvector('english', coalesce(new.abstract,'')), 'B');
  return new;
END
$$ LANGUAGE plpgsql;


CREATE TRIGGER tsvectorupdate BEFORE INSERT OR UPDATE
ON papers FOR EACH ROW EXECUTE FUNCTION papers_search_trigger();

CREATE INDEX idx_search_vector ON papers USING GIN (search_vector);
