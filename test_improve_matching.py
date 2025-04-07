#!/usr/bin/env python
# Test for improve_matching.py

import unittest
from improve_matching import (
    create_fallback_analysis, 
    DOMAIN_KEYWORDS, 
    DOMAIN_RELATIONS,
    DOMAIN_SKILLS,
    format_results
)

class TestImproveMatching(unittest.TestCase):
    """Test the improve_matching.py functionality"""
    
    def test_fallback_analysis(self):
        """Test the fallback analysis with different CV texts"""
        
        # Test case 1: IT/Networks domain
        cv_text_it = """
        Ingénieur en développement logiciel avec 5 ans d'expérience en Python, JavaScript et React.
        Compétences en DevOps avec Docker et Kubernetes. Expérience dans le développement d'API REST
        et d'applications web. Certification AWS Solutions Architect.
        """
        result_it = create_fallback_analysis(cv_text_it)
        self.assertEqual('informatique_reseaux', result_it['main_domain'])
        self.assertGreater(len(result_it['skills']), 0)
        
        # Test case 2: Finance domain
        cv_text_finance = """
        Analyste financier avec 8 ans d'expérience en contrôle de gestion et comptabilité.
        Expertise en analyse budgétaire, reporting financier et audit interne.
        Maîtrise des outils SAP Finance et Excel avancé.
        """
        result_finance = create_fallback_analysis(cv_text_finance)
        self.assertEqual('finance', result_finance['main_domain'])
        
        # Test case 3: Industrial Automation domain
        cv_text_auto = """
        Ingénieur automaticien spécialisé en programmation d'automates Siemens et Schneider.
        Développement de solutions SCADA et mise en place de systèmes de supervision industrielle.
        Expérience en robotique et systèmes embarqués.
        """
        result_auto = create_fallback_analysis(cv_text_auto)
        self.assertEqual('automatismes_info_industrielle', result_auto['main_domain'])
    
    def test_format_results(self):
        """Test the formatting of job results"""
        # Mock job offers
        job_offers = [
            {
                "id": "job1",
                "score": 0.75,
                "metadata": {
                    "title": "Développeur Python",
                    "company": "Tech Solutions",
                    "location": "Paris",
                    "domain": "informatique_reseaux"
                },
                "content": "Nous recherchons un développeur Python expérimenté."
            },
            {
                "id": "job2",
                "score": 0.65,
                "metadata": {
                    "title": "Data Engineer",
                    "company": "Data Corp",
                    "location": "Lyon",
                    "domain": "informatique_reseaux"
                },
                "content": "Poste de Data Engineer pour projet Big Data."
            }
        ]
        
        # Format results with matching domain
        formatted = format_results(job_offers, "informatique_reseaux")
        
        # Check if results are properly formatted
        self.assertEqual(len(formatted), 2)
        self.assertEqual(formatted[0]["domain_match"], "✅")
        self.assertEqual(formatted[0]["score"], 0.75 + 0.2)  # Score + domain bonus
        
        # Test with non-matching domain
        formatted_non_match = format_results(job_offers, "finance")
        self.assertEqual(formatted_non_match[0]["domain_match"], "❌")
        self.assertEqual(formatted_non_match[0]["score"], 0.75)  # No domain bonus
    
    def test_domain_keywords(self):
        """Test the domain keywords definitions"""
        # Check that all required domains are defined
        required_domains = [
            "informatique_reseaux", 
            "automatismes_info_industrielle", 
            "finance", 
            "genie_civil_btp", 
            "genie_industriel"
        ]
        
        for domain in required_domains:
            self.assertIn(domain, DOMAIN_KEYWORDS)
            self.assertGreater(len(DOMAIN_KEYWORDS[domain]), 5)  # Should have at least 5 keywords
            
        # Check domain relations consistency
        for domain, related in DOMAIN_RELATIONS.items():
            self.assertIn(domain, DOMAIN_KEYWORDS)
            for rel in related:
                self.assertIn(rel, DOMAIN_KEYWORDS)
                
        # Check domain skills
        for domain in DOMAIN_SKILLS:
            self.assertIn(domain, DOMAIN_KEYWORDS)
            self.assertGreater(len(DOMAIN_SKILLS[domain]), 0)

if __name__ == '__main__':
    print("=" * 70)
    print("TESTING IMPROVE_MATCHING.PY")
    print("=" * 70)
    unittest.main() 