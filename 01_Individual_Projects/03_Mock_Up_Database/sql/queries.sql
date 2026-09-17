-- QUERIES
-- SCHEMA: test_results 
-- (sample_id, test_type, result_value, unit, run_date, operator_id, site_code, status)


-- Query 1: Total number of tests per site
SELECT site_code, COUNT(*) AS "Total_Num_Test"
FROM public.test_results
GROUP BY site_code


-- Query 2: Average result value per test type (passed only)
SELECT test_type, AVG(result_value)
FROM public.test_results
WHERE status='passed'
GROUP BY test_type


-- Query 3: Operator(s) with most tests at Vimodrone (VIM)
SELECT operator_id, COUNT(*) AS total_tests
FROM public.test_results
WHERE site_code='VIM'
GROUP BY operator_id
ORDER BY total_tests DESC
LIMIT 5


-- Query 4: Site(s) where more than 20% of tests failed
SELECT site_code
FROM public.test_results
WHERE ((COUNT(status))/(COUNT(STATUS)) *100%) > 20%
GROUP BY site_code

-- Query 5: For each site, test type with highest failed tests


-- Query 6: Operator(s) who performed tests at all three sites