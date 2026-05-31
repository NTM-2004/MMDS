CREATE TABLE Audio (
    audioId SERIAL PRIMARY KEY,   
    fileName VARCHAR(255),        
    filePath TEXT,
    speakerFeature vector(16) 
);
CREATE TABLE Keyword (
    keywordId SERIAL PRIMARY KEY,
    word VARCHAR(255) UNIQUE
);

CREATE TABLE InvertedFile (
    audioId INT,                  
    keywordId INT,                
    tf_idfScore REAL,
    PRIMARY KEY(audioId, keywordId),
    FOREIGN KEY (audioId) REFERENCES Audio(audioId),
    FOREIGN KEY (keywordId) REFERENCES Keyword(keywordId)
);

CREATE INDEX idx_keyword_search ON InvertedFile(keywordId);
