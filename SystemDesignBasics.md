layout: page
title: "System Design Basics"
permalink: /basic/

Scalability
What is it that we really mean by scalability? 
A service is said to be scalable if when we increase the resources in a system, it results in increased performance in a manner proportional to resources added. 
Increasing performance in general means serving more units of work, but it can also be to handle larger units of work, such as when datasets grow.

In distributed systems there are other reasons for adding resources to a system; for example to improve the reliability of the offered service. Introducing redundancy is an important first line of defense against failures. An always-on service is said to be scalable if adding resources to facilitate redundancy does not result in a loss of performance.
