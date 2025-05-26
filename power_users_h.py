import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load ratings
ratings = pd.read_csv('./data/ratings.csv')

# Get top 10 users with most ratings
user_rating_counts = ratings['userId'].value_counts().nlargest(1) # 2698 largest
top_users_df = user_rating_counts.reset_index()
top_users_df.columns = ['userId', 'num_ratings']

# Sort userId as ordered categorical for horizontal barplot
top_users_df['userId'] = top_users_df['userId'].astype(str)
top_users_df['userId'] = pd.Categorical(
    top_users_df['userId'],
    categories=top_users_df.sort_values('num_ratings', ascending=True)['userId'],
    ordered=True
)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_users_df,
    x='num_ratings',
    y='userId',
    palette='viridis',
    legend=False,
    hue="userId"
)

# Dashed vertical line at shortest bar (min num_ratings)
min_count = top_users_df['num_ratings'].min()
plt.axvline(x=min_count, linestyle='--', color='gray')

# Annotate the line on the x-axis
plt.text(min_count + 1, 0, f'{min_count} ratings', color='gray', ha='left', va='center')

# Titles and labels
plt.title('Top 10 Power Users by Number of Ratings (Horizontal)')
plt.xlabel('Number of Ratings')
plt.ylabel('User ID')
plt.tight_layout()
plt.show()
