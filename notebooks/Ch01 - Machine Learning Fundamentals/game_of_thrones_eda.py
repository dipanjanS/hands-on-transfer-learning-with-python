
# coding: utf-8

# ## Import required packages
import numpy as np
import pandas as pd
from collections import Counter

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

# setting params
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

sns.set_style('whitegrid')
sns.set_context('talk')

plt.rcParams.update(params)


# ## Load Dataset
# 
# In this step we load the ```battles.csv``` for analysis
# load dataset
battles_df = pd.read_csv('battles.csv')


# Display sample rows
print(battles_df.head())


# ## Explore raw properties
print("Number of attributes available in the dataset = {}".format(battles_df.shape[1]))


# View available columns and their data types
print(battles_df.dtypes)


# Analyze properties of numerical columns
battles_df.describe()

# ## Number of Battles Fought
# This data is till **season 5** only

print("Number of battles fought={}".format(battles_df.shape[0]))


# ## Battle Distribution Across Years
# The plot below shows that maximum bloodshed happened in the year 299 with 
# a total of 20 battles fought!

sns.countplot(y='year',data=battles_df)
plt.title('Battle Distribution over Years')
plt.show()


# ## Which Regions saw most Battles?

sns.countplot(x='region',data=battles_df)
plt.title('Battles by Regions')
plt.show()


# ### Death or Capture of Main Characters by Region

# No prizes for guessing that Riverlands have seen some of the main characters 
# being killed or captured. Though _The Reach_ has seen 2 battles, none of the 
# major characters seemed to have fallen there.
f, ax1 = plt.subplots()
ax2 = ax1.twinx()
temp_df = battles_df.groupby('region').agg({'major_death':'sum',
                                            'major_capture':'sum'}).reset_index()
temp_df.loc[:,'dummy'] = 'dummy'
sns.barplot(x="region", y="major_death", 
            hue='dummy', data=temp_df, 
            estimator = np.sum, ax = ax1, 
            hue_order=['dummy','other'])

sns.barplot(x="region", y="major_capture", 
            data=temp_df, hue='dummy',
            estimator = np.sum, ax = ax2, 
            hue_order=['other','dummy'])

ax1.legend_.remove()
ax2.legend_.remove()


# ## Who Attacked the most?
# The Baratheon boys love attacking as they lead the pack with 38% while
# Rob Stark has been the attacker in close second with 27.8% of the battles.
attacker_king = battles_df.attacker_king.value_counts()
attacker_king.name='' # turn off annoying y-axis-label
attacker_king.plot.pie(figsize=(6, 6),autopct='%.2f')


# ## Who Defended the most?
# Rob Stark and Baratheon boys are again on the top of the pack. Looks like 
# they have been on either sides of the war lot many times.

defender_king = battles_df.defender_king.value_counts()
defender_king.name='' # turn off annoying y-axis-label
defender_king.plot.pie(figsize=(6, 6),autopct='%.2f')


# ## Battle Style Distribution
# Plenty of battles all across, yet the men of Westeros and Essos are men of honor. 
# This is visible in the distribution which shows **pitched battle** as the 
# most common style of battle.

sns.countplot(y='battle_type',data=battles_df)
plt.title('Battle Type Distribution')
plt.show()


# ## Attack or Defend?
# Defending your place in Westeros isn't easy, this is clearly visible from 
# the fact that 32 out of 37 battles were won by attackers

sns.countplot(y='attacker_outcome',data=battles_df)
plt.title('Attack Win/Loss Distribution')
plt.show()


# ## Winners
# Who remembers losers? (except if you love the Starks)
# The following plot helps us understand who won how many battles and how, 
# by attacking or defending.

attack_winners = battles_df[battles_df.\
                            attacker_outcome=='win']\
                                ['attacker_king'].\
                                value_counts().\
                                reset_index()
                                
attack_winners.rename(
        columns={'index':'king',
                 'attacker_king':'wins'},
         inplace=True)

attack_winners.loc[:,'win_type'] = 'attack'

defend_winners = battles_df[battles_df.\
                            attacker_outcome=='loss']\
                            ['defender_king'].\
                            value_counts().\
                            reset_index()
defend_winners.rename(
        columns={'index':'king',
                 'defender_king':'wins'},
         inplace=True)

defend_winners.loc[:,'win_type'] = 'defend'                                                                     


sns.barplot(x="king", 
            y="wins", 
            hue="win_type", 
            data=pd.concat([attack_winners,
                            defend_winners]))
plt.title('Kings and Their Wins')
plt.ylabel('wins')
plt.xlabel('king')
plt.show()


# ## Battle Commanders
# A battle requires as much brains as muscle power. 
# The following is a distribution of the number of commanders involved on attacking and defending sides.

battles_df['attack_commander_count'] = battles_df.\
                                        dropna(subset=['attacker_commander']).\
                                        apply(lambda row: \
                                              len(row['attacker_commander'].\
                                                  split()),axis=1)
battles_df['defend_commander_count'] = battles_df.\
                                        dropna(subset=['defender_commander']).\
                                        apply(lambda row: \
                                              len(row['defender_commander'].\
                                                  split()),axis=1)

battles_df[['attack_commander_count',
            'defend_commander_count']].plot(kind='box')


# ## How many houses fought in a battle?
# Were the battles evenly balanced? The plots tell the whole story.
battles_df['attacker_house_count'] = (4 - battles_df[['attacker_1', 
                                                'attacker_2', 
                                                'attacker_3', 
                                                'attacker_4']].\
                                        isnull().sum(axis = 1))

battles_df['defender_house_count'] = (4 - battles_df[['defender_1',
                                                'defender_2', 
                                                'defender_3', 
                                                'defender_4']].\
                                        isnull().sum(axis = 1))

battles_df['total_involved_count'] = battles_df.apply(lambda row: \
                                      row['attacker_house_count'] + \
                                      row['defender_house_count'],
                                                      axis=1)
battles_df['bubble_text'] = battles_df.apply(lambda row: \
          '{} had {} house(s) attacking {} house(s) '.\
          format(row['name'],
                 row['attacker_house_count'],
                 row['defender_house_count']),
                 axis=1)


# ## Unbalanced Battles
# Most battles so far have seen more houses forming alliances while attacking. 
# There are only a few friends when you are under attack!

house_balance = battles_df[
        battles_df.attacker_house_count != \
        battles_df.defender_house_count][['name',
                                       'attacker_house_count',
                                       'defender_house_count']].\
        set_index('name')
house_balance.plot(kind='bar')


# ## Battles and The size of Armies
# Attackers don't take any chances, they come in huge numbers, keep your eyes open

army_size_df = battles_df.dropna(subset=['total_involved_count',
                          'attacker_size',
                          'defender_size',
                         'bubble_text'])
army_size_df.plot(kind='scatter', x='defender_size',y='attacker_size',
                  s=army_size_df['total_involved_count']*150)


# ## Archenemies?
# The Stark-Baratheon friendship has taken a complete U-turn with a total of 19 battles and counting. Indeed there is no one to be trusted in this land.

temp_df = battles_df.dropna(
                    subset = ["attacker_king",
                              "defender_king"])[
                                ["attacker_king",
                                 "defender_king"]
                                ]

archenemy_df = pd.DataFrame(
                list(Counter(
                        [tuple(set(king_pair)) 
                         for king_pair in temp_df.values
                         if len(set(king_pair))>1]).
                            items()),
                columns=['king_pair',
                         'battle_count'])

archenemy_df['versus_text'] = archenemy_df.\
                                apply(
                                    lambda row:
                                '{} Vs {}'.format(
                                        row[
                                            'king_pair'
                                            ][0],
                                        row[
                                            'king_pair'
                                            ][1]),
                                        axis=1)
archenemy_df.sort_values('battle_count',
                         inplace=True,
                         ascending=False)


archenemy_df[['versus_text',
              'battle_count']].set_index('versus_text',
                                          inplace=True)
sns.barplot(data=archenemy_df,
            x='versus_text',
            y='battle_count')
plt.xticks(rotation=45)
plt.xlabel('Archenemies')
plt.ylabel('Number of Battles')
plt.title('Archenemies')
plt.show()