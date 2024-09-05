import pstats

# Load the profile data
p = pstats.Stats('output.prof')

# Sort the data by time spent
p.sort_stats('cumulative')

# Print the top 10 lines that took the most time
p.print_stats(500)