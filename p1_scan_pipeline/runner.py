from handwriting_system.p1_scan_pipeline import ScanPipeline
pipeline = ScanPipeline()
pipeline.process_with_wordlist("dataset/scans/session1/page1.png")
# or for a whole folder:
pipeline.process_session("dataset/scans/session1/")
