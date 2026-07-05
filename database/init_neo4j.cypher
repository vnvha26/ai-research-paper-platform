CREATE CONSTRAINT paper_id_unique FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE;

CREATE CONSTRAINT author_id_unique FOR (a:Author) REQUIRE a.author_id IS UNIQUE;